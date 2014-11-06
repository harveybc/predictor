<?php
$this->breadcrumbs = array(
	'Motores',
	Yii::t('app', 'Index'),
);

$this->menu=array(
	array('label'=>Yii::t('app', 'Crear') . ' Informe', 'url'=>array('create')),
	array('label'=>Yii::t('app', 'Gestionar') . ' Informes', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Motores'); ?>
<div class="forms100c" style="text-align:left;">
<?php $this->widget('zii.widgets.CListView', array(
	'dataProvider'=>$dataProvider,
	'itemView'=>'_view',
)); ?>
</div>