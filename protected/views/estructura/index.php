<?php
$this->breadcrumbs = array(
	'Equipos',
	Yii::t('app', 'Index'),
);

$this->menu=array(
	//array('label'=>Yii::t('app', 'Crear') . ' Equipos', 'url'=>array('create')),
	array('label'=>Yii::t('app', 'Gestionar') . ' Equipos', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Equipos'); ?>
<div class="forms100c" style="text-align:left;">
<?php $this->widget('zii.widgets.CListView', array(
	'dataProvider'=>$dataProvider,
	'itemView'=>'_view',
)); ?>
</div>