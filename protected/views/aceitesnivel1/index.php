<?php
$this->breadcrumbs = array(
	'Lubricantes',
	Yii::t('app', 'Index'),
);

$this->menu=array(
	array('label'=>Yii::t('app', 'Crear') . ' Lubricantes', 'url'=>array('create')),
	array('label'=>Yii::t('app', 'Gestionar') . ' Lubricantes', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Lista de Mediciondes de Lubricantes'); ?>
<div class="forms100c" style="text-align:left;">
<?php $this->widget('zii.widgets.CListView', array(
	'dataProvider'=>$dataProvider,
	'itemView'=>'_view',
)); ?>
</div>
