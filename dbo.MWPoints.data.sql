Select * from dbo.MWPoints where PointName IN (Select PointName from dbo.ThresholdPoints where PointID = '67')
--Select * From dbo.VIEW_T_ZB_EA where INSTR_NO = 'C06-09'
--select * From dbo.VIEW_T_ZB_EA_AS where pname = '9'
--select * from dbo.View_T_ZB_MX_AS where pname = '67'