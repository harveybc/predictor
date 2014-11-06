<?php
$this->breadcrumbs = array(
	'Informes de Tableros Eléctricos',
	Yii::t('app', 'Index'),
);

$this->menu=array(
	array('label'=>Yii::t('app', 'Crear') . ' Informe', 'url'=>array('create')),
	array('label'=>Yii::t('app', 'Gestionar') . ' Informes', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Informes de Tableros Eléctricos'); ?>
<div class="forms100c" style="text-align:left;">
<?php $this->widget('zii.widgets.CListView', array(
	'dataProvider'=>$dataProvider,
	'itemView'=>'_view',
)); ?>
</div>